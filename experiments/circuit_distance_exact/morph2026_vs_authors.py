r"""morph2026's standard schedule against the authors' builder, gate by gate (as the lane's morph_vs_authors.py).

    python -I morph2026_vs_authors.py --clone <clone of github.com/Mac-Shaw/morphing_qec_circuits>

For every standard-schedule target in morph2026_targets.json (and the lane's two morph codes as a control) it calls the
authors' BBParams(l, m, a_list, b_list).get_standard_morphing_specification("x") -- which handles any |A|, |B| >= 2
on Z_l x Z_m -- and compares the CNOT layers of both contraction rounds, the X / Z measurement lists and each
mid-cycle stabilizer's declared contraction round against the round our gates contract it in. The clone is
untrusted: run with -I; galois / gurobipy / surface_sim are replaced by the lane's stubs (morph_vs_authors.STUBS).
A deliberately corrupted schedule (two layers swapped) is run as a negative control and must FAIL.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import tempfile

sys.dont_write_bytecode = True  # -I with an untrusted clone on sys.path: write nothing next to it
HERE = pathlib.Path(__file__).resolve().parent
LANE = HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True)
    a = ap.parse_args()
    sys.path[:0] = [str(HERE), str(LANE)]
    import morph_vs_authors as mva  # read-only: STUBS
    import morph2026 as M
    import bb_circuits as bc
    stubs = pathlib.Path(tempfile.mkdtemp(prefix="m2026_stubs_"))
    for name, text in mva.STUBS.items():
        (stubs / name).parent.mkdir(parents=True, exist_ok=True)
        (stubs / name).write_text(text)
    sys.path[:0] = [str(stubs), str(pathlib.Path(a.clone).resolve())]
    from morphing_qec_circuits.bb_params import BBParams

    cases = [(k, M.lane_code(k)) for k in bc.MORPH_CODES]
    cases += [(k, M.spec(k)["code"]) for k, t in M.TARGETS.items() if t["kind"] == "std"]

    def compare(code, gates, meas):
        params = BBParams(code.l, code.m, [list(e) for e in code.A], [list(e) for e in code.B])
        if not params.valid_standard_morphing_parameters("x"):
            return dict(valid=False)
        sp = params.get_standard_morphing_specification("x")
        same_rounds = True
        for p in "XZ":
            for g, declared in enumerate(sp.stabiliser_contraction_rounds[p]):
                rest = [bool(bc._contract(set(sp.mid_cycle_stabilisers[p][g]), p, gates[r]) - set(meas[p][r]))
                        for r in (0, 1)]
                same_rounds &= rest.count(False) == 1 and declared == [rest.index(False)]
        return dict(valid=True,
                    gates_identical=[[sorted(map(tuple, x)) for x in rnd] for rnd in sp.gate_lists]
                    == [[sorted(x) for x in rnd] for rnd in gates],
                    measurements_identical=all(sorted(sp.measurement_lists[p][r]) == sorted(meas[p][r])
                                               for p in "XZ" for r in (0, 1)),
                    contraction_rounds_identical=bool(same_rounds),
                    cnots=sum(len(x) for rnd in gates for x in rnd))

    for name, code in cases:
        gates, meas, _, _ = M.morphing_schedule(code, (1, 0))
        print(json.dumps(dict(target=name, l=code.l, A=code.A, B=code.B, **compare(code, gates, meas))))
    # negative control: swap two layers of round 0 of the first target; the comparison must fail
    name, code = cases[2]
    gates, meas, _, _ = M.morphing_schedule(code, (1, 0))
    bad = copy.deepcopy(gates)
    bad[0][0], bad[0][1] = bad[0][1], bad[0][0]
    print(json.dumps(dict(target=name + " (layers 0,1 of round 0 swapped: must FAIL)", **compare(code, bad, meas))))


if __name__ == "__main__":
    main()
