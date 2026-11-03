r"""Paper 1's IBM run, re-certified under the epsilon-checker with every receipt kept.

    python experiments/ibm_live/ibm_epsilon_rerun.py --fetch      # once: the job's raw registers, into evidence
    python experiments/ibm_live/ibm_epsilon_rerun.py              # certify, decide twice, keep every receipt

WHY (paper 1, third referee, 2026-11-02, M1). The repetition-code memory run on ibm_fez (job d9nt2imij12s73fu9ihg,
preregistered, d = 9, 5 rounds, 4,000 shots) was certified on 2026-08-02 by the production exact checker of its day,
which never tested that a packing's sets are T'-odd -- it accepted forged optimality proofs (src/oneq/dual_forgery.py,
PARITY_FORGERIES) until 2026-10-18 -- and its certificates were not kept. Its "2,296 of 2,296" therefore rests on a
checker later found unsound. This re-runs it the way every headline rate of the paper was produced:
  * --fetch reads the completed job's registers (no QPU time) and commits them, so the run is replayable from the
    repository; the control is that they reproduce the original count of nontrivial shots exactly;
  * the same uniform declared graph and the same producer call as the original (`certify_hard`, the full ladder);
  * every receipt decided by check_epsilon (eps_max = 1e-6) AND by the separately written scaled-integer verifier of
    tools/replay_epsilon_receipts.py, which must agree on status, slack and bound;
  * the forgery battery of the epsilon rerun on every shot with a packing, each forgery re-verified false first;
  * the commit and the git bytes of every module that produced or decided a verdict are recorded.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ibm_live"))
sys.path.insert(0, str(ROOT / "tools"))
JOB = "d9nt2imij12s73fu9ihg"
EV = ROOT / "evidence" / "ibm_live"
SHOTS = EV / f"{JOB}.shots.json"
OUT = EV / f"{JOB}.epsilon.json"
RECEIPTS = EV / f"{JOB}.epsilon_receipts.jsonl.gz"
EPS_MAX = Fraction(1, 10**6)
JUDGED = ["src/oneq/certifying_decoder.py", "src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py",
          "src/oneq/forgeries.py", "experiments/ibm_live/rep_code_cert.py", "tools/replay_epsilon_receipts.py",
          "experiments/ibm_live/ibm_epsilon_rerun.py"]


def _pack(a: np.ndarray) -> str:
    return base64.b64encode(np.packbits(a.astype(np.uint8), axis=None).tobytes()).decode()


def _unpack(s: str, shape) -> np.ndarray:
    n = int(np.prod(shape))
    return np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))[:n].reshape(shape)


def fetch() -> int:
    from qiskit_ibm_runtime import QiskitRuntimeService
    rec = json.loads((EV / f"{JOB}.job.json").read_text(encoding="utf-8"))
    d, r = rec["config"]["distance"], rec["config"]["rounds"]
    res = QiskitRuntimeService().job(JOB).result()[0]
    regs = {f"m{t}": np.array(getattr(res.data, f"m{t}").to_bool_array(), dtype=np.uint8) for t in range(r)}
    regs["fin"] = np.array(res.data.fin.to_bool_array(), dtype=np.uint8)
    doc = {"schema": "oneq-ibm-raw-registers/1", "job_id": JOB, "backend": rec["backend"], "distance": d,
           "rounds": r, "note": "the completed job's classical registers as returned by IBM (to_bool_array), read "
                                "once on 2026-11-02 with no QPU time; packed row-major with numpy.packbits",
           "registers": {k: {"shape": list(v.shape), "packed_b64": _pack(v)} for k, v in regs.items()}}
    doc["sha256_unpacked"] = hashlib.sha256(b"".join(regs[k].tobytes() for k in sorted(regs))).hexdigest()
    SHOTS.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {SHOTS.relative_to(ROOT) if SHOTS.is_relative_to(ROOT) else SHOTS}: {regs['fin'].shape[0]} shots")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        return fetch()
    import pymatching
    from oneq.certifying_decoder import BOUNDARY, CertifyingDecoder
    from oneq.forgeries import assert_all_false, forgeries
    from oneq.matching_cert import MatchingCertificate
    from oneq.matching_cert_epsilon import check_epsilon
    from rep_code_cert import decode_graph, detectors_from_records
    from replay_epsilon_receipts import verify_b

    raw = json.loads(SHOTS.read_text(encoding="utf-8"))
    regs = {k: _unpack(v["packed_b64"], v["shape"]) for k, v in raw["registers"].items()}
    if hashlib.sha256(b"".join(regs[k].tobytes() for k in sorted(regs))).hexdigest() != raw["sha256_unpacked"]:
        raise SystemExit("the committed registers do not unpack to the bytes they were hashed as")
    d, r = raw["distance"], raw["rounds"]
    shots = regs["fin"].shape[0]
    det3 = np.zeros((shots, r + 1, d - 1), dtype=np.uint8)
    for s in range(shots):
        det3[s] = detectors_from_records(np.stack([regs[f"m{t}"][s] for t in range(r)]), regs["fin"][s])
    edges = decode_graph(d, r)
    cd = CertifyingDecoder(edges)
    m = pymatching.Matching()
    for (u, v, w) in edges:
        if v == BOUNDARY:
            m.add_boundary_edge(int(u), weight=w)
        else:
            m.add_edge(int(u), int(v), weight=w)
    original = json.loads((EV / f"{JOB}.certified.json").read_text(encoding="utf-8"))
    status, forg, forg_acc, bogus, lines = Counter(), Counter(), Counter(), 0, []
    largest = Fraction(0)
    nontrivial = 0
    for s in range(shots):
        fired = [int(i) for i in np.flatnonzero(det3[s].reshape(-1))]
        if not fired:
            continue
        nontrivial += 1
        syn = np.zeros((r + 1) * (d - 1), dtype=np.uint8)
        syn[fired] = 1
        corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY)) for u, v in m.decode_to_edges_array(syn))
        res = cd.certify_hard(fired, corr)
        duals = dict(res.cert.blossom_duals) if res.cert is not None else dict(res.best_dual or {})
        cert = MatchingCertificate(matched_edges=corr, node_potentials={}, blossom_duals=duals)
        va = check_epsilon(edges=edges, syndrome=fired, cert=cert, eps_max=EPS_MAX)
        listed = [(tuple(sorted(int(x) for x in S)), float(z)) for S, z in duals.items() if z > 0]
        vb = verify_b(edges, fired, list(corr), listed, EPS_MAX)
        if va.status != vb[0] or (va.status != "INVALID_INPUT" and (va.epsilon, va.lower_bound) != vb[1:]):
            raise SystemExit(f"shot {s}: the two verifiers disagree ({va.status} vs {vb[0]})")
        status[va.status] += 1
        if va.accepted and va.epsilon > largest:
            largest = va.epsilon
        lines.append(json.dumps({"shot": s, "fired": fired, "correction": [list(e) for e in corr],
                                 "duals": [[list(S), float.hex(z)] for S, z in listed], "status": va.status,
                                 "epsilon": None if va.epsilon is None else str(va.epsilon),
                                 "lower_bound": None if va.lower_bound is None else str(va.lower_bound)},
                                separators=(",", ":")))
        if duals:
            battery = forgeries(cd.W, fired, corr, duals, res.primal)
            some = next(iter(duals))
            battery += [("N_nan_dual", MatchingCertificate(corr, {}, {**duals, some: math.nan}), "non-finite"),
                        ("N_inf_dual", MatchingCertificate(corr, {}, {**duals, some: math.inf}), "non-finite")]
            bogus += len(assert_all_false(cd.W, fired, res.primal, [b for b in battery if b[2] != "non-finite"]))
            for nm, fc, _why in battery:
                forg[nm] += 1
                forg_acc[nm] += check_epsilon(edges=edges, syndrome=fired, cert=fc, eps_max=EPS_MAX).accepted
    if nontrivial != original["nontrivial"] or shots != original["shots"]:
        raise SystemExit(f"the fetched registers give {nontrivial} nontrivial of {shots}, not the original "
                         f"{original['nontrivial']} of {original['shots']}: not the same data")
    body = ("\n".join(lines) + "\n").encode()
    RECEIPTS.write_bytes(gzip.compress(body, 6, mtime=0))

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    accepted = status.get("EXACT_OPTIMAL", 0) + status.get("EPSILON_OPTIMAL", 0)
    out = {"schema": "oneq-ibm-epsilon-rerun/1", "job_id": JOB, "backend": raw["backend"], "distance": d,
           "rounds": r, "shots": shots, "nontrivial": nontrivial, "accepted": accepted, "status": dict(status),
           "eps_max": str(EPS_MAX), "largest_accepted_epsilon": str(largest),
           "largest_accepted_epsilon_float": float(largest), "verifiers_agree_on": nontrivial,
           "forgeries_attempted": sum(forg.values()), "forgeries_accepted": sum(forg_acc.values()),
           "forgeries_by_class": dict(forg), "battery_attacks_not_false": bogus,
           "original": {"certified": original["certified"], "nontrivial": original["nontrivial"],
                        "checker": "the production exact checker of 2026-08-02, which did not test T'-oddness"},
           "producer": "CertifyingDecoder(uniform declared graph).certify_hard, the full ladder, as the original",
           "receipts": RECEIPTS.relative_to(ROOT).as_posix() if RECEIPTS.is_relative_to(ROOT) else str(RECEIPTS),
           "receipts_sha256": hashlib.sha256(body).hexdigest(),
           "head": {"commit": git("rev-parse", "HEAD"),
                    "uncommitted_changes_in_src_oneq": bool(git("status", "--porcelain", "--", "src/oneq")),
                    "sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                               for p in JUDGED}}}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{accepted} of {nontrivial} accepted ({dict(status)}); both verifiers agree on all; forgeries "
          f"{out['forgeries_accepted']}/{out['forgeries_attempted']} accepted; largest eps {float(largest):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
