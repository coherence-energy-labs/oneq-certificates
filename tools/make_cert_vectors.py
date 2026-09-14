r"""Build cross-language conformance vectors for the decoding certificate.

AGREEMENT IS THE CONFORMANCE DEFINITION. Two independent checkers reaching the
same verdict on the same input is evidence; one checker passing its own tests
is not. The Python checker has been wrong four times in ways its own suite did
not see -- it assumed PyMatching returned a matching on syndrome nodes when it
returns paths, it accepted node potentials as a T-join dual, it mishandled the
virtual boundary, and it scored equal-weight alternative optima as forgeries.
Each was a misreading of the problem. A port would have inherited all four,
which is why the JavaScript is written from the spec and pinned here.

THE VECTORS INCLUDE REFUSALS, and mostly refusals. A suite of certificates
that all verify tests one branch: the happy path. It would not notice a second
implementation that accepts everything -- which is the failure mode that
matters, since a checker that never refuses is worthless and looks perfect.
So every forgery in the battery becomes a vector, and the JavaScript has to
refuse each one too.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.cert_format import f2h, graph_fingerprint, to_document  # noqa: E402
from oneq.certifying_decoder import CertifyingDecoder             # noqa: E402
from oneq.forgeries import assert_all_false, forgeries            # noqa: E402
from oneq.matching_cert import BOUNDARY, check                    # noqa: E402


def collect(edges, shots, tag, out):
    cd = CertifyingDecoder(edges)
    fp = graph_fingerprint(edges)
    graph = [[int(u), int(v), f2h(w)] for u, v, w in edges]
    for i, (syndrome, correction) in enumerate(shots):
        r = cd.certify(syndrome, correction)
        doc = to_document(edges=edges, syndrome=syndrome, cert=r.cert,
                          proven_optimal=r.accepted, primal=r.primal,
                          bound=r.bound, reason=r.reason, graph_sha256=fp)
        out.append({"name": f"{tag}-{i}-genuine", "graph": graph,
                    "document": doc, "expected_accepted": r.accepted,
                    "expected_reason": r.reason})
        if r.cert is None:
            continue
        battery = forgeries(cd.W, syndrome, correction, r.cert.blossom_duals,
                            r.primal)
        bogus = assert_all_false(cd.W, syndrome, r.primal, battery)
        if bogus:
            raise SystemExit(f"battery emitted a non-attack: {bogus}")
        for name, forged, why in battery:
            v = check(edges=edges, syndrome=syndrome, cert=forged)
            if v.accepted:
                raise SystemExit(f"the Python checker ACCEPTED {name} -- fix "
                                 f"that before publishing vectors")
            fdoc = to_document(edges=edges, syndrome=syndrome, cert=forged,
                               proven_optimal=True,      # the forgery CLAIMS it
                               primal=cd.weight(forged.matched_edges),
                               bound=sum(forged.blossom_duals.values()),
                               reason=f"forged: {name} ({why})",
                               graph_sha256=fp)
            out.append({"name": f"{tag}-{i}-{name}", "graph": graph,
                        "document": fdoc, "expected_accepted": False,
                        "expected_reason": v.reason})


def main() -> int:
    vectors: list[dict] = []

    # A graph small enough to reason about by hand, so a disagreement can be
    # diagnosed rather than merely observed.
    tiny = [(0, 1, 1.0), (1, 2, 1.0), (0, BOUNDARY, 3.0), (2, BOUNDARY, 3.0),
            (1, BOUNDARY, 2.5)]
    collect(tiny, [([0, 2], ((0, 1), (1, 2))),
                   ([0], ((0, BOUNDARY),)),
                   ([0, 2], ((0, BOUNDARY), (2, BOUNDARY)))], "tiny", vectors)

    # A real surface-code shot: weights are irrational-looking doubles, which
    # is precisely where a decimal round-trip would betray a ported checker.
    try:
        import pymatching
        import stim
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=3, rounds=3,
            after_clifford_depolarization=5e-3,
            before_measure_flip_probability=5e-3,
            after_reset_flip_probability=5e-3,
            before_round_data_depolarization=5e-3)
        m = pymatching.Matching.from_detector_error_model(
            circ.detector_error_model(decompose_errors=True))
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
                  float(e[2].get("weight", 1.0))) for e in m.edges()]
        cd = CertifyingDecoder(edges)
        det, _ = circ.compile_detector_sampler(seed=4242).sample(
            40, separate_observables=True)
        shots = []
        for s in range(40):
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            shots.append((fired, tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                                       for a, b in m.decode_to_edges_array(det[s]))))
            if len(shots) >= 6:
                break
        collect(edges, shots, "surface-d3", vectors)
    except ImportError:
        print("stim/pymatching absent: surface-code vectors skipped",
              file=sys.stderr)

    # A vector that only a SECOND IMPLEMENTATION could have produced. Python
    # deserialised the dual into a dict (a repeated member set silently kept
    # the last value); the JavaScript summed the array. On the same bytes
    # Python read an objective of 2.0 and ACCEPTED where the JavaScript read
    # 2.4 and refused. Neither reading was right -- the document does not
    # denote one certificate -- and both now refuse it. Pinned here so the
    # divergence cannot reopen.
    if vectors:
        acc = next(v for v in vectors if v["expected_accepted"]
                   and v["document"].get("dual"))
        dup = json.loads(json.dumps(acc))
        dup["name"] = "ambiguous-duplicate-dual-set"
        first = dup["document"]["dual"][0]
        dup["document"]["dual"] = [[first[0], f2h(0.25)]] + dup["document"]["dual"]
        dup["expected_accepted"] = False
        dup["expected_reason"] = "AMBIGUOUS DOCUMENT: member set appears more than once"
        vectors.append(dup)

    # A vector the fingerprint must catch: the right certificate, the wrong
    # graph. Nothing about the arithmetic is wrong here -- only the binding is.
    if vectors:
        base = next(v for v in vectors if v["expected_accepted"])
        wrong = json.loads(json.dumps(base))
        wrong["name"] = "graph-substitution"
        wrong["graph"] = [[0, 1, f2h(0.5)], [1, 2, f2h(0.5)],
                          [0, BOUNDARY, f2h(3.0)], [2, BOUNDARY, f2h(3.0)]]
        wrong["expected_accepted"] = False
        wrong["expected_reason"] = "GRAPH MISMATCH"
        vectors.append(wrong)

    n_acc = sum(1 for v in vectors if v["expected_accepted"])
    doc = {
        "schema": "oneq-certificate-conformance/1",
        "note": ("Each vector records the PYTHON checker's verdict. A second "
                 "implementation conforms when it reaches the same verdict on "
                 "every one. Mostly refusals on purpose: a suite where "
                 "everything verifies would not notice a checker that accepts "
                 "everything."),
        "float_encoding": ("every double is its exact 64-bit big-endian "
                           "pattern in hex; decimal JSON is not bit-identical "
                           "across languages"),
        "counts": {"total": len(vectors), "expected_accept": n_acc,
                   "expected_refuse": len(vectors) - n_acc},
        "vectors": vectors,
    }
    dest = ROOT / "conformance" / "cert_vectors.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{len(vectors)} vectors ({n_acc} accept, {len(vectors)-n_acc} refuse) "
          f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
