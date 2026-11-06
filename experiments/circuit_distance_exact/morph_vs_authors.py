r"""R198: the lane's morphing schedule against the authors' own builder, gate by gate.

    python morph_vs_authors.py --clone <path to a clone of github.com/Mac-Shaw/morphing_qec_circuits>

Imports the authors' `BBParams.get_standard_morphing_specification` (arXiv 2407.16336) and compares, for both
codes of `bb_circuits.MORPH_CODES` and every homomorphism (x, y, xy): the CNOT layers of both contraction rounds,
the X / Z measurement lists, and each mid-cycle stabilizer's contraction round (theirs DECLARED, ours DERIVED by
carrying the stabilizer through the layers). Their modules import galois, gurobipy and surface_sim at import
time for things the builder never calls; those three are replaced by stubs written to a temporary directory, so
the comparison needs only numpy and stim. The clone is untrusted input: its modules define functions and
constants only (read before first use, 2026-10-20, commit 1ee705f); run this with `python -I`. JSON lines on
stdout, nothing under evidence/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

STUBS = {
    "galois.py": "def GF(p):\n    return lambda x: x\n",
    "gurobipy.py": "GRB = None\n",
    "surface_sim/__init__.py": "",
    "surface_sim/models.py": "class CircuitNoiseModel:\n    pass\n\n\nclass NoiselessModel:\n    pass\n",
    "surface_sim/setup.py": "class Setup:\n    pass\n",
}


def compare(clone: pathlib.Path) -> list[dict]:
    stubs = pathlib.Path(tempfile.mkdtemp(prefix="r198_stubs_"))
    for name, text in STUBS.items():
        (stubs / name).parent.mkdir(parents=True, exist_ok=True)
        (stubs / name).write_text(text, encoding="utf-8", newline="\n")  # temporary stubs; LF (serialization.py)
    sys.path[:0] = [str(stubs), str(clone), str(HERE)]
    from morphing_qec_circuits.bb_params import BBParams

    import bb_circuits as bc
    out = []
    for kind, (l, m, a_terms, b_terms) in bc.MORPH_CODES.items():
        el = lambda t: [t[1], 0] if t[0] == "x" else [0, t[1]]  # noqa: E731
        for hom in bc.MORPH_HOMS:
            params = BBParams(l, m, [el(t) for t in a_terms], [el(t) for t in b_terms])
            if not params.valid_standard_morphing_parameters(hom):
                out.append(dict(kind=kind, hom=hom, valid=False))
                continue
            spec = params.get_standard_morphing_specification(hom)
            gates, meas = bc.morphing_schedule(l, m, a_terms, b_terms, hom)
            same_rounds = True
            for p in "XZ":
                for g, declared in enumerate(spec.stabiliser_contraction_rounds[p]):
                    rest = [bool(bc._contract(set(spec.mid_cycle_stabilisers[p][g]), p, gates[r]) - set(meas[p][r]))
                            for r in (0, 1)]
                    same_rounds &= rest.count(False) == 1 and declared == [rest.index(False)]
            out.append(dict(
                kind=kind, hom=hom, valid=True,
                gates_identical=[[sorted(map(tuple, x)) for x in rnd] for rnd in spec.gate_lists]
                == [[sorted(x) for x in rnd] for rnd in gates],
                measurements_identical=all(sorted(spec.measurement_lists[p][r]) == sorted(meas[p][r])
                                           for p in "XZ" for r in (0, 1)),
                contraction_rounds_identical=same_rounds, cnots=sum(len(x) for rnd in gates for x in rnd)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True)
    for row in compare(pathlib.Path(ap.parse_args().clone).resolve()):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
