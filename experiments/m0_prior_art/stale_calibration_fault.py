r"""A fault nobody constructed to be caught: the decoder's graph goes stale.

THE OBJECTION THIS ANSWERS. The seeded-fault gate mutates a decoder's OUTPUT,
and a fair reviewer will say the mutants were chosen by someone who knew what
the certificate checks -- that "syndrome-consistent but heavier" is caught by a
weight check is close to definitional. The reply cannot be an argument. It has
to be a fault that was not designed against the detector.

SO THIS INJECTS THE FAULT UPSTREAM, INTO THE DECODE PATH. The decoder runs
unmodified, on a graph built from a DRIFTED error model -- the calibration it
was configured with no longer matches the device it is decoding. Nothing about
the correction is touched; PyMatching returns whatever its own algorithm
returns, exactly as it would in production. The certificate is then checked
against the TRUE graph.

WHY THIS IS THE REAL FAILURE MODE, not a contrived one. Decoding weights come
from a calibration snapshot. Devices drift between calibrations, calibration
jobs fail and the previous snapshot is reused, a config file points at the
wrong run, a deployment ships last month's priors. Every one of those produces
exactly this: a decoder confidently solving the wrong optimisation problem,
returning corrections that explain the syndrome perfectly and are not
minimum-weight for the device in front of it.

AND IT IS INVISIBLE TO EVERYTHING CURRENTLY DEPLOYED. The correction is
syndrome-consistent by construction -- the decoder did its job correctly on the
graph it was given -- so a syndrome check passes every time. The logical error
rate rises slightly, which is indistinguishable from the device having a worse
day. That is the whole problem: the failure looks like physics.

WHAT IS MEASURED. For a sweep of drift magnitudes: how often the stale decoder
departs from the true optimum, how far, how much of it syndrome consistency
notices (structurally: none), and how much the certificate notices. Drift of
zero is the control -- at zero drift the stale decoder IS the true decoder, and
any "defect" reported there is the harness lying.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder   # noqa: E402
from oneq.matching_cert import BOUNDARY                 # noqa: E402


def drifted_dem(dem, factor: float, rng):
    """The same error model with its probabilities multiplied by drift.

    Each mechanism's probability is scaled by a random factor in
    [1/factor, factor]. That is what a stale snapshot looks like: the graph's
    STRUCTURE is right -- same detectors, same mechanisms -- and only the
    weights are wrong, which is precisely why nothing downstream notices.
    """
    import re

    import stim
    # BUILT BY TEXT, not via DetectorErrorModel.append. The binding rejects a
    # re-supplied argument list for DECOMPOSED errors -- the ones carrying `^`
    # separators -- and those are exactly the instructions that must survive,
    # because dropping a decomposition changes the matching graph's structure
    # rather than only its weights, which is a different fault from the one
    # being injected. The DEM text format round-trips exactly, so rewriting the
    # probability in each `error(p) ...` line preserves everything else.
    lines = []
    for inst in dem.flattened():
        text = str(inst)
        if inst.type != "error" or factor <= 1.0:
            lines.append(text)
            continue
        p0 = float(inst.args_copy()[0])
        mult = float(np.exp(rng.uniform(-np.log(factor), np.log(factor))))
        p1 = min(max(p0 * mult, 1e-12), 0.49)
        lines.append(re.sub(r"^error\([^)]*\)", "error(" + repr(p1) + ")",
                            text, count=1))
    return stim.DetectorErrorModel("\n".join(lines))


def main() -> int:
    import pymatching
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--drifts", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=a.distance, rounds=a.rounds,
        after_clifford_depolarization=a.p,
        before_measure_flip_probability=a.p,
        after_reset_flip_probability=a.p,
        before_round_data_depolarization=a.p)
    true_dem = circ.detector_error_model(decompose_errors=True)
    true_m = pymatching.Matching.from_detector_error_model(true_dem)
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in true_m.edges()]
    cd = CertifyingDecoder(edges)
    det, obs = circ.compile_detector_sampler(seed=20260731).sample(
        a.shots, separate_observables=True)

    rows = []
    print(f"d={a.distance} r={a.rounds}  {a.shots} shots\n")
    print(f"{'drift':>7} {'syndrome-consistent':>21} {'suboptimal':>12} "
          f"{'syndrome catches':>17} {'certificate catches':>20} {'mean excess':>12}")
    for factor in a.drifts:
        rng = np.random.default_rng(7)
        stale_m = pymatching.Matching.from_detector_error_model(
            drifted_dem(true_dem, factor, rng))

        n = consistent = suboptimal = cert_caught = 0
        excess = []
        logical_true = logical_stale = 0
        for s in range(a.shots):
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            # the decoder runs UNMODIFIED, on the stale graph
            try:
                corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                             for u, v in stale_m.decode_to_edges_array(det[s]))
            except Exception:                         # noqa: BLE001
                continue
            if any(k not in cd.W for k in corr):
                continue                              # not expressible in the true graph
            n += 1
            # DETECTOR 1: does the correction explain the syndrome? It always
            # will -- the decoder solved a correct problem with wrong weights.
            deg: dict[int, int] = {}
            for (u, v) in corr:
                for x in (u, v):
                    if x != BOUNDARY:
                        deg[x] = deg.get(x, 0) + 1
            if {k for k, c in deg.items() if c % 2 == 1} == set(fired):
                consistent += 1
            # DETECTOR 2: the certificate, checked against the TRUE graph
            r = cd.certify(fired, corr)
            true_corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                              for u, v in true_m.decode_to_edges_array(det[s]))
            true_w = cd.weight(true_corr)
            if r.primal > true_w + 1e-9:
                suboptimal += 1
                excess.append(r.primal - true_w)
                if not r.accepted:
                    cert_caught += 1
            logical_true += int(bool(true_m.decode(det[s])[0]) != bool(obs[s][0]))
            logical_stale += int(bool(stale_m.decode(det[s])[0]) != bool(obs[s][0]))

        row = {"drift_factor": factor, "shots": n,
               "syndrome_consistent": consistent,
               "genuinely_suboptimal": suboptimal,
               "syndrome_check_caught": n - consistent,
               "certificate_caught": cert_caught,
               "mean_excess_weight": float(np.mean(excess)) if excess else 0.0,
               "logical_errors_true_model": logical_true,
               "logical_errors_stale_model": logical_stale}
        rows.append(row)
        print(f"{factor:7.2f} {consistent:>13}/{n:<7} {suboptimal:12d} "
              f"{n-consistent:17d} {cert_caught:20d} "
              f"{row['mean_excess_weight']:12.4f}")

    ctrl = rows[0] if rows and rows[0]["drift_factor"] == 1.0 else None
    print()
    if ctrl:
        print(f"CONTROL (drift 1.0): {ctrl['genuinely_suboptimal']} suboptimal "
              f"corrections -- must be 0, or the harness is inventing defects")
    worst = rows[-1]
    print(f"AT DRIFT {worst['drift_factor']}: syndrome consistency caught "
          f"{worst['syndrome_check_caught']} of {worst['genuinely_suboptimal']} "
          f"genuinely suboptimal corrections; the certificate caught "
          f"{worst['certificate_caught']}")
    print(f"  logical errors: {worst['logical_errors_true_model']} with the true "
          f"model, {worst['logical_errors_stale_model']} with the stale one "
          f"-- the damage a rate benchmark would have to resolve")

    out = {"schema": "oneq-stale-calibration-fault/1",
           "distance": a.distance, "rounds": a.rounds, "p": a.p,
           "shots": a.shots, "rows": rows,
           "why_this_is_not_a_constructed_fault": (
               "the decoder runs UNMODIFIED and its output is never touched. "
               "The fault is upstream, in the graph it was configured with -- "
               "a stale calibration snapshot, which is a real and documented "
               "operational failure, not a mutant designed against the "
               "detector it is being scored by."),
           "why_nothing_deployed_notices": (
               "the correction is syndrome-consistent BY CONSTRUCTION: the "
               "decoder solved a correct problem with wrong weights. The only "
               "downstream symptom is a slightly higher logical error rate, "
               "which is indistinguishable from the device having a worse day. "
               "The failure looks like physics."),
           "control": ("drift 1.0 must produce zero suboptimal corrections; a "
                       "nonzero count there means the harness is manufacturing "
                       "defects rather than detecting them")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / f"stale_calibration_d{a.distance}.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nevidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
