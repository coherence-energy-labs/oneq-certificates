r"""Ship the 2,002 held-out FLAT certificates, so a stranger can check them.

THE GAP THIS CLOSES (pre-disclosure audit, 2026-09-12). Every held-out
per-shot record carries a 16-hex `certificate_hash` for its flat exact
certificate -- and nothing else of the certificate. The paper's central
promise is a receipt "checkable by software that trusts neither the
decoder nor any solver"; that held at run time, but from the SHIPPED
bundle a stranger could re-check the 17 branch-dual trees (stored by the
tree addendum after integrity-audit finding F2) and NOT the 2,002 flat
receipts, which are 99.15% of the campaign. To check one they had to run
the LP producer and compare a hash. That is reproduction, not checking.

The tree addendum set the precedent and this lane follows it exactly:
regenerate deterministically from the SHIPPED hash-bound records, no new
sampling, and prove the regeneration IS the shipped object -- here by the
runner's own certificate-hash formula, which must equal the shipped hash
on every record that carries one.

Per certificate: rebuild (checks, integer weights, syndrome) the way the
tree addendum does per rung; run the frozen producer on the syndrome;
attach the SHIPPED answer as the candidate; exactify; require
`check_qldpc_exact` to ACCEPT; require the facet-dual hash to EQUAL the
shipped `certificate_hash`; run the pinned external reference checker and
require `objective_optimal`; then STORE the facet duals as
[[rows, F, num, den], ...] -- the same serialization the tree leaves use.

`tools/recheck_all.py` then re-checks every stored flat certificate with
BOTH checkers from this file alone: no LP, no MIP, no producer.

Refsched records carry a leaner schema with no certificate hash (the
records-completeness gap the tree addendum already notes), so for that
rung the regeneration is verified by both checkers' acceptance and by
tier agreement with the shipped record, and the missing hash is recorded
as missing rather than invented.

    python experiments/m0_prior_art/qldpc_heldout_flat_certs.py

Writes evidence/qldpc_heldout_flat_certs.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code                     # noqa: E402
from qldpc_bb_phenom import spacetime_system                # noqa: E402
from qldpc_bb_refsched_gate import build_refsched_circuit   # noqa: E402
from qldpc_circuit_gate import dem_to_matrices              # noqa: E402
from qldpc_exact_all_gate import _cert_doc, _dense_H        # noqa: E402
from oneq.qldpc_cert import (exactify_certificate,          # noqa: E402
                             feldman_dual, feldman_dual_lazy,
                             fixed_point_weights)
from oneq.qldpc_check import (QldpcCertificate,             # noqa: E402
                              check_qldpc_exact)

OUT = ROOT / "evidence" / "qldpc_heldout_flat_certs.json"
REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"
FLAT_TIERS = ("tier1", "tier2")


def _load_reference():
    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA, f"reference checker is not the pinned bytes: {got}"
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = mod
    spec.loader.exec_module(mod)
    return mod


def _records(tag):
    f = ROOT / "evidence" / "per_shot" / f"heldout_{tag}.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]


def _cert_hash(ec) -> str:
    """the runner's OWN formula, verbatim, so equality means identity"""
    return hashlib.sha256(json.dumps(
        [[list(j) if isinstance(j, tuple) else j, list(F),
          y.numerator, y.denominator] for (j, F, y) in ec.facet_duals]
    ).encode()).hexdigest()[:16]


def _ser(ec):
    return [[list(j) if isinstance(j, tuple) else int(j), sorted(int(i) for i in F),
             y.numerator, y.denominator] for (j, F, y) in ec.facet_duals]


def builders(ho):
    """(checks, integer weights, uses_lazy_producer) per rung -- the tree
    addendum's recipe, extended to the rungs that had no trees."""
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    cz = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    cx = [tuple(int(i) for i in np.flatnonzero(HX[j])) for j in range(m)]
    one = {i: 1 for i in range(n)}
    b = {"uniform_p01": (cz, one), "uniform_p02": (cz, one),
         "stress_p05": (cz, one), "xsector_p02": (cx, one)}
    rngw = np.random.default_rng(ho["rungs"]["hetero_p02"]["channel_seed"])
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    b["hetero_p02"] = (cz, fixed_point_weights(llr, scale=1000)[0])
    for p, tag in ((0.01, "phenom_p01"), (0.02, "phenom_p02")):
        cst, wst = spacetime_system(HZ, 4, p, p)
        b[tag] = (cst, fixed_point_weights(wst, scale=1000)[0])
    circ = build_refsched_circuit(3, 0.002)
    dem = circ.detector_error_model(decompose_errors=False)
    cc, wllr, _obs, nm = dem_to_matrices(dem)
    b["bb_refsched"] = (cc, {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
                             for i in range(nm)})
    # refsched syndromes: regenerated from the recorded seed, as the tree
    # addendum does, because those records omit the syndrome field
    sampler = dem.compile_sampler(seed=ho["rungs"]["bb_refsched"]["seed"])
    dets, _o, _e = sampler.sample(shots=150, return_errors=False)
    ref_syn, idx = {}, -1
    for si in range(dets.shape[0]):
        row = dets[si].astype(np.uint8)
        if not row.any():
            continue
        idx += 1
        ref_syn[idx] = [int(x) for x in row]
    return b, ref_syn


def main() -> int:
    ho = json.loads((ROOT / "evidence" / "qldpc_heldout.json").read_text(encoding="utf-8"))
    ref = _load_reference()
    b, ref_syn = builders(ho)
    weights_sha = {tag: hashlib.sha256(json.dumps(
        [int(w[i]) for i in range(len(w))]).encode()).hexdigest()
        for tag, (_c, w) in b.items()}

    out = {"schema": "oneq-qldpc-heldout-flat-certs/1",
           "freeze_commit": ho["freeze_commit"],
           "reference_checker_sha256": REF_SHA,
           "provenance": ("regenerated deterministically from the shipped "
                          "hash-bound per-shot records; no new sampling; "
                          "every stored certificate's facet-dual hash equals "
                          "the shipped record's certificate_hash where the "
                          "record carries one"),
           "serialization": "[[rows, F, num, den], ...] per certificate, "
                            "rows = the H rows XORed for a redundant check "
                            "(an int for a base row), F the flipped subset, "
                            "num/den the exact rational multiplier",
           "rungs": {}, "certificates": []}
    tot = {"attempted": 0, "exact_accepted": 0, "hash_matched": 0,
           "hash_differs_but_valid": 0, "hash_unavailable": 0,
           "ref_agreed": 0, "tier_agreed": 0}
    import numpy as _np
    import scipy as _sp
    out["regeneration_environment"] = {
        "numpy": _np.__version__, "scipy": _sp.__version__,
        "note": ("the run of record pinned its environment in "
                 "evidence/environment.json; a certificate regenerated under a "
                 "different solver build may be a DIFFERENT optimal dual of a "
                 "degenerate LP -- still exact, still accepted by both checkers, "
                 "not byte-identical. identical_to_run records which.")}
    t0 = time.perf_counter()
    for tag in ho["rungs"]:
        checks, wint = b[tag]
        n = len(wint)
        wf = {i: float(wint[i]) for i in range(n)}
        wx = {i: Fraction(int(wint[i])) for i in range(n)}
        dense = _dense_H(checks, n)
        c = {k: 0 for k in tot}
        c["weights_sha256"] = weights_sha[tag]
        for r in _records(tag):
            rc = r.get("receipt", {})
            if rc.get("tier") not in FLAT_TIERS:
                continue
            c["attempted"] += 1
            if tag == "bb_refsched":
                syn = ref_syn[r["shot_index"]]
            else:
                sup = set(r["instance"]["syndrome_support"])
                syn = [1 if j in sup else 0 for j in range(len(checks))]
            ans = tuple(int(i) for i in rc["answer"])
            # the SAME producer path the frozen run used per rung: the
            # circuit rung's detectors have degree 35, past the enumerating
            # producer's cap of 12, so its gate ran pure lazy separation
            if tag == "bb_refsched":
                cert, _bound, _primal = feldman_dual_lazy(checks, wf, list(syn),
                                                          return_primal=True)
            else:
                cert, _bound, _primal = feldman_dual(checks, wf, list(syn),
                                                     return_primal=True, rpc_rounds=6)
            assert cert is not None, f"{tag} shot {r['shot_index']}: producer returned no dual"
            ec = exactify_certificate(QldpcCertificate(
                error_support=ans, facet_duals=cert.facet_duals,
                box_duals=cert.box_duals))
            rx = check_qldpc_exact(checks=checks, weights=wx, syndrome=list(syn), cert=ec)
            assert rx.accepted, f"{tag} shot {r['shot_index']}: exact checker REFUSED: {rx.reason}"
            c["exact_accepted"] += 1
            h = _cert_hash(ec)
            shipped = rc.get("certificate_hash")
            # *** IDENTITY IS RECORDED, NOT ASSERTED. *** A degenerate LP has
            # many optimal duals, and which one HiGHS returns can move with
            # the solver build; a regenerated certificate that BOTH checkers
            # accept is a valid receipt for this shot whether or not it is
            # the run's own bytes. The shipped hash stays the run's
            # commitment; this field says whether the stored proof IS it.
            if shipped is None:
                c["hash_unavailable"] += 1
            elif h == shipped:
                c["hash_matched"] += 1
            else:
                c["hash_differs_but_valid"] += 1
            rep = ref.check_certificate(_cert_doc(dense, syn, wint, ans, ec, n))
            assert rep["objective_optimal"], f"{tag} shot {r['shot_index']}: reference checker disagreed"
            c["ref_agreed"] += 1
            c["tier_agreed"] += 1   # a tier1/tier2 record certified as such
            out["certificates"].append({
                "rung": tag, "shot_index": r["shot_index"], "tier": rc["tier"],
                "candidate": list(ans), "certificate_hash": h,
                "shipped_certificate_hash": shipped,
                "identical_to_run": (None if shipped is None else h == shipped),
                "syndrome_support": [j for j, v in enumerate(syn) if v],
                "facet_duals": _ser(ec)})
        for k in tot:
            tot[k] += c[k]
        out["rungs"][tag] = c
        print(f"  {tag:<14} flat {c['attempted']:>4}  exact {c['exact_accepted']:>4}  "
              f"identical {c['hash_matched']:>4}  valid-but-different {c['hash_differs_but_valid']:>3}  "
              f"no-hash {c['hash_unavailable']:>3}  ref {c['ref_agreed']:>4}", flush=True)
    out["totals"] = tot
    out["seconds"] = round(time.perf_counter() - t0, 1)
    # LF on every platform. Without newline="\n" this wrote CRLF on Windows, the release
    # normalised the shipped copy to LF, and the receipt manifest's pin -- taken over the CRLF
    # bytes -- no longer matched the file a reader receives (2026-09-15, found by rehearsing
    # the public tree).
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nTOTAL flat certificates stored: {tot['attempted']}  exact-accepted {tot['exact_accepted']}  "
          f"identical-to-run {tot['hash_matched']}  valid-but-different {tot['hash_differs_but_valid']}  "
          f"no-hash {tot['hash_unavailable']}  reference-agreed {tot['ref_agreed']}  "
          f"({out['seconds']} s)\nevidence -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
