r"""Paper 2's flat receipts, decided by a checker in a second language.

Both checkers that replay paper 2's receipts are Python. This exports every flat
receipt named by the pinned receipt manifest, with its hash-bound instance and
integer weight vector, adds planted inputs derived from real receipts, runs
verifier-js/verify-qldpc-flat.mjs (JavaScript, exact BigInt rationals, written
from the manuscript's statement of the problem), and requires

  * every receipt accepted;
  * every planted defect refused FOR ITS OWN REASON -- a defect refused by some
    other rule would not show that the rule it targets is alive;
  * two lattice-boundary inputs per dataset decided exactly: a valid, overload-free,
    tight witness scaled so its bound is U - 1 (must be refused) and U - 1/2
    (must be accepted), which pins the acceptance rule itself.

Every planted input is also given to the production checker, which must reach
the same verdict, so a mutation that leaves a true claim true cannot count as a
defect.

    python tools/qldpc_js_agreement.py              # writes evidence/qldpc_js_agreement.json
    python tools/qldpc_js_agreement.py --js other.mjs --no-write
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from oneq.qldpc_check import QldpcCertificate, check_qldpc_exact  # noqa: E402
from replay_receipts import PINNED_MANIFEST_SHA256               # noqa: E402

MANIFEST = ROOT / "evidence" / "receipt_manifest.json"
FLAT = ROOT / "evidence" / "qldpc_heldout_flat_certs.json"
JS = ROOT / "verifier-js" / "verify-qldpc-flat.mjs"
OUT = ROOT / "evidence" / "qldpc_js_agreement.json"
SAFE = 2 ** 53

# the refusal each planted defect must receive (a substring of the checker's reason)
REASON = {
    "candidate_extra_variable": "does not satisfy check",
    "candidate_variable_out_of_range": "out of range",
    "syndrome_bit_dropped": "does not satisfy check",
    "infeasible_empty_candidate": "does not satisfy check",
    "facet_variable_outside_check": "outside its check",
    "facet_wrong_parity": "wrong parity",
    "facet_listed_twice": "listed twice",
    "negative_multiplier": "negative multiplier",
    "zero_denominator": "zero denominator",
    "row_out_of_range": "out of range",
    "combined_check_with_one_row": "at least two rows",
    "combined_check_parity_is_xor": "wrong parity",
    "lattice_boundary_refuse": "not proven",
}


class AgreementError(Exception):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_batch() -> dict:
    raw = MANIFEST.read_bytes()
    if _sha(raw) != PINNED_MANIFEST_SHA256:
        raise AgreementError("receipt manifest does not match the pinned hash")
    man = json.loads(raw)
    for rel, h in man["files"].items():
        if rel.startswith("evidence/instances/") and _sha((ROOT / rel).read_bytes()) != h:
            raise AgreementError(f"{rel} does not match its manifest hash")
    flat = {f"heldout_flat/{c['rung']}/{c['shot_index']}": c
            for c in json.loads(FLAT.read_text(encoding="utf-8"))["certificates"]}
    instances = {k: json.loads((ROOT / v["file"]).read_text(encoding="utf-8")) for k, v in man["instances"].items()}
    batch = {"instances": {k: {"n": v["n_variables"], "checks": v["checks"]} for k, v in instances.items()},
             "weights": {}, "receipts": []}
    for m in man["receipts"]:
        if m["cohort"] != "heldout_flat":
            continue
        c = flat[m["id"]]
        wf = m["weights_file"]
        if wf not in batch["weights"]:
            vec = [int(x) for x in json.loads((ROOT / wf).read_text(encoding="utf-8"))["vector"]]
            if _sha(json.dumps(vec).encode()) != m["weights_vector_sha256"]:
                raise AgreementError(f"{wf} does not match its manifest hash")
            batch["weights"][wf] = vec
        batch["receipts"].append({"id": m["id"], "instance": m["instance"], "weights": wf,
                                  "syndrome_support": c["syndrome_support"], "candidate": c["candidate"],
                                  "facet_duals": c["facet_duals"], "expect": "accept", "kind": "receipt"})
    if len(batch["receipts"]) != man["expected_counts"]["heldout_flat"]:
        raise AgreementError("the batch does not hold every flat receipt the manifest names")
    return batch


def _python_accepts(batch: dict, r: dict) -> bool:
    """The production checker's verdict on the same input (unconstructible input is a refusal)."""
    inst = batch["instances"][r["instance"]]
    checks = [tuple(c) for c in inst["checks"]]
    syn = [0] * len(checks)
    try:
        for j in r["syndrome_support"]:
            syn[int(j)] = 1
        fd = [(tuple(row) if isinstance(row, list) else int(row), tuple(F), Fraction(int(a), int(b)))
              for (row, F, a, b) in r["facet_duals"]]
    except (IndexError, ZeroDivisionError, ValueError, TypeError):
        return False
    wx = {i: Fraction(w) for i, w in enumerate(batch["weights"][r["weights"]])}
    cert = QldpcCertificate(error_support=tuple(int(i) for i in r["candidate"]), facet_duals=fd, box_duals=())
    return check_qldpc_exact(checks=checks, weights=wx, syndrome=syn, cert=cert).accepted


def _tight_no_overload(batch: dict, r: dict):
    """(U) if the receipt's witness has no overloaded variable and a bound equal to U, else None."""
    inst = batch["instances"][r["instance"]]
    checks, w = inst["checks"], batch["weights"][r["weights"]]
    s = set(r["syndrome_support"])
    load: dict[int, Fraction] = {}
    by = Fraction(0)
    for row, F, a, b in r["facet_duals"]:
        y = Fraction(a, b)
        if y == 0:
            continue
        rows = row if isinstance(row, list) else [row]
        support: dict[int, int] = {}
        for rr in rows:
            for i in checks[rr]:
                support[i] = support.get(i, 0) ^ 1
        Fs = set(F)
        for i, odd in support.items():
            if odd:
                load[i] = load.get(i, Fraction(0)) + (-y if i in Fs else y)
        by += y * (1 - len(F))
    if any(t > w[i] for i, t in load.items()):
        return None
    U = sum(w[i] for i in r["candidate"])
    _ = s
    return U if by == U and U >= 2 else None


def _scaled(r: dict, num: int, den: int) -> list:
    out = []
    for row, F, a, b in r["facet_duals"]:
        na, nb = a * num, b * den
        g = math.gcd(na, nb) or 1
        na, nb = na // g, nb // g
        if abs(na) >= SAFE or nb >= SAFE:
            raise AgreementError("a scaled multiplier left the exact-integer range of the wire format")
        out.append([copy.deepcopy(row), list(F), na, nb])
    return out


def planted(batch: dict) -> list[dict]:
    """Defects derived from the first receipt of each dataset, and lattice-boundary inputs."""
    by_rung: dict[str, list[dict]] = {}
    for r in batch["receipts"]:
        by_rung.setdefault(r["id"].split("/")[1], []).append(r)
    out = []
    for rung, rs in sorted(by_rung.items()):
        base = rs[0]
        inst = batch["instances"][base["instance"]]
        n, m = inst["n"], len(inst["checks"])
        pos = next((k for k, f in enumerate(base["facet_duals"]) if f[2] > 0), None)

        def variant(cls, mutate, src=base, expect="refuse"):
            r = copy.deepcopy(src)
            r["id"] = f"planted/{cls}/{rung}"
            r["expect"] = expect
            r["kind"] = cls
            mutate(r)
            out.append(r)

        free = next(i for i in range(n) if i not in set(base["candidate"]))
        variant("candidate_extra_variable", lambda r: r["candidate"].append(free))
        variant("candidate_variable_out_of_range", lambda r: r["candidate"].append(n))
        if base["syndrome_support"]:
            variant("syndrome_bit_dropped", lambda r: r["syndrome_support"].pop(0))
            variant("infeasible_empty_candidate", lambda r: (r["candidate"].clear(), r["facet_duals"].clear()))
        if pos is not None:
            f0 = base["facet_duals"][pos]
            rows = f0[0] if isinstance(f0[0], list) else [f0[0]]
            support: dict[int, int] = {}
            for row in rows:
                for i in inst["checks"][row]:
                    support[i] = support.get(i, 0) ^ 1
            outside = next(i for i in range(n) if not support.get(i))
            inside = next((i for i in sorted(support) if support[i] and i not in f0[1]), None)

            def set_facet(r, k, value):
                r["facet_duals"][pos][k] = value

            variant("facet_variable_outside_check", lambda r: set_facet(r, 1, sorted(f0[1] + [outside])))
            if f0[1] or inside is not None:
                variant("facet_wrong_parity", lambda r: set_facet(r, 1, f0[1][:-1] if f0[1] else [inside]))
            variant("facet_listed_twice", lambda r: r["facet_duals"].append(copy.deepcopy(f0)))
            variant("negative_multiplier", lambda r: set_facet(r, 2, -f0[2]))
            variant("zero_denominator", lambda r: set_facet(r, 3, 0))
            variant("row_out_of_range", lambda r: set_facet(r, 0, m))
            if not isinstance(f0[0], list):
                variant("combined_check_with_one_row", lambda r: set_facet(r, 0, [f0[0]]))
        # two fired checks combine to parity 0 over GF(2) (1 under OR): an empty F is then the
        # wrong parity, so this input is refused only by a checker that folds parities by XOR
        fired2 = next((r for r in rs if len(r["syndrome_support"]) >= 2), None)
        if fired2 is not None:
            j0, k0 = sorted(fired2["syndrome_support"])[:2]
            variant("combined_check_parity_is_xor",
                    lambda r: r["facet_duals"].append([[j0, k0], [], 1, 2]), src=fired2)
        tight = next(((r, U) for r in rs if (U := _tight_no_overload(batch, r)) is not None), None)
        if tight is not None:
            src, U = tight
            variant("lattice_boundary_refuse",
                    lambda r: r.__setitem__("facet_duals", _scaled(src, U - 1, U)), src=src)
            variant("lattice_boundary_accept",
                    lambda r: r.__setitem__("facet_duals", _scaled(src, 2 * U - 1, 2 * U)), src=src, expect="accept")
    return out


def run(js: pathlib.Path = JS, write: bool = True) -> dict:
    if shutil.which("node") is None:
        raise AgreementError("node is not installed")
    batch = load_batch()
    extra = planted(batch)
    wrong_truth = [x["id"] for x in extra if _python_accepts(batch, x) != (x["expect"] == "accept")]
    if wrong_truth:
        raise AgreementError(f"planted inputs the production checker decides the other way: {wrong_truth[:5]}")
    batch["receipts"] += extra
    with tempfile.TemporaryDirectory() as tmp:
        bp = pathlib.Path(tmp) / "batch.json"
        bp.write_text(json.dumps(batch), encoding="utf-8")
        proc = subprocess.run(["node", str(js), str(bp)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=3600)
    verdict = {}
    for ln in proc.stdout.splitlines():
        if ln and not ln.startswith("SUMMARY"):
            parts = ln.split("\t")
            verdict[parts[0]] = (parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")
    missing = [x["id"] for x in batch["receipts"] if x["id"] not in verdict]
    if missing:
        raise AgreementError(f"no verdict for {len(missing)} inputs (node exit {proc.returncode}): {missing[:3]}")
    wrong = []
    for x in batch["receipts"]:
        got, why = verdict[x["id"]]
        want = "ACCEPT" if x["expect"] == "accept" else "REFUSE"
        if got != want or (want == "REFUSE" and REASON[x["kind"]] not in why):
            wrong.append({"id": x["id"], "verdict": got, "reason": why})
    receipts = [x for x in batch["receipts"] if x["kind"] == "receipt"]
    defects = [x for x in extra if x["expect"] == "refuse"]
    doc = {
        "schema": "oneq-qldpc-js-agreement/2",
        "checker": "verifier-js/verify-qldpc-flat.mjs",
        "checker_sha256": _sha(pathlib.Path(js).read_bytes()),
        "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
        "manifest_sha256": PINNED_MANIFEST_SHA256,
        "flat_receipts": len(receipts),
        "flat_receipts_accepted": sum(1 for x in receipts if verdict[x["id"]][0] == "ACCEPT"),
        "planted_defects": len(defects),
        "planted_defects_refused_for_their_reason": sum(
            1 for x in defects if verdict[x["id"]][0] == "REFUSE" and REASON[x["kind"]] in verdict[x["id"]][1]),
        "lattice_boundary_accepts": sum(1 for x in extra if x["kind"] == "lattice_boundary_accept"
                                        and verdict[x["id"]][0] == "ACCEPT"),
        "planted_by_class": dict(Counter(x["kind"] for x in extra)),
        "disagreements": wrong[:50],
        "command": "python tools/qldpc_js_agreement.py",
    }
    if write:
        OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    if wrong:
        raise AgreementError(f"{len(wrong)} disagreements; first: {wrong[:3]}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--js", default=str(JS))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    try:
        doc = run(pathlib.Path(a.js), write=not a.no_write)
    except AgreementError as exc:
        print(f"JS AGREEMENT FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: doc[k] for k in ("flat_receipts", "flat_receipts_accepted", "planted_defects",
                                          "planted_defects_refused_for_their_reason",
                                          "lattice_boundary_accepts", "node")}))
    print("JS AGREEMENT PASSED: every flat receipt accepted, every planted defect refused for its own reason, "
          "and the lattice boundary decided exactly, by a checker in a second language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
