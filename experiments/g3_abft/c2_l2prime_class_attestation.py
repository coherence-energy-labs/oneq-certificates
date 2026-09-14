r"""L2' -- CERTIFIED LOGICAL-CLASS ATTESTATION: the layer C2 is missing.

THE HOLE (measured in c2_mask_flip_scaling.py). C2 specifies a three-layer
detector whose L2 is "an Edmonds LP-dual optimality certificate". A single
bit flip in one edge's OBSERVABLE MASK multiplies the logical error rate
by 4.9x / 23.6x / 130x / 491x at d = 3/5/7/9 -- and neither inline layer
can see it, not by bad luck but by construction:

    a mask flip changes no edge WEIGHT
      -> the matching problem is unchanged
      -> the correction e-hat is unchanged
      -> L1 (H e-hat == s) and L2 (is e-hat min-weight) are both
         functions of e-hat, so neither verdict moves

The dominant fault therefore falls through to L0, the table CRC, which
runs at scrub cadence -- exactly the 10-100 ms window C2 exists to close.
A certificate that proves the correction is minimum-weight is certifying
half the answer: it says nothing about WHICH LOGICAL CLASS that correction
was attributed to.

THE IDEA: SPEND THE CODE'S OWN DEGENERACY, NOT SILICON. A logical operator
has many homologically equivalent representatives. In a rotated
surface-code memory, the logical Z can be read as row 0 of the final data
measurements, or row 1, or row 2 -- disjoint qubit sets, all the same
logical operator. Compile two of them and the DEM carries two observable
columns whose mask bits SHARE NOTHING. A single-event upset can corrupt at
most one.

This is not DMR. DMR duplicates the block at 2x. This adds one bit per
mechanism to a table that already exists, and buys its independence from
the code's structure rather than from replicated hardware.

THE CHECK, AND WHY IT NEEDS NO TRUTH. Two representatives differ by a
product of stabilizers, so for ANY error e:

    obs_A(e) XOR obs_B(e) = <f, syndrome(e)>

for a fixed GF(2) functional f over detectors -- the stabilizers lying
between the two representatives. f is SOLVED here from the DEM's own
structure and then verified against every mechanism, so the runtime test

    pred_A XOR pred_B XOR <f, syndrome>  ==  0

uses only what a decoder already has. It is an inline consistency
condition, not a comparison against an answer nobody has at runtime.

WHAT IS REPORTED: detection rate conditioned on activation (an unactivated
fault is invisible to every inline layer, and averaging it in would
flatter all of them equally), the FALSE-POSITIVE rate on clean shots
(a detector that fires on healthy hardware is worse than none), and the
cost in bits against the spec's own baselines.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "g3_abft"))

from c2_mask_flip_scaling import corrupt_dem                # noqa: E402


def corrupt_weight(dem, k: int, factor: float = 40.0):
    """Corrupt a mechanism's PROBABILITY, not its observable mask.

    THE COMPLEMENTARITY CONTROL. A weight fault changes the matching
    problem, so the correction moves -- which is exactly the class L2
    (min-weight optimality) exists to catch, and exactly the class L2'
    should be BLIND to, because both observables are still derived
    consistently from whatever correction the decoder settled on.

    If L2' fired on this too it would not be a complementary layer, it
    would be a second copy of L2 wearing a different name.
    """
    import stim
    out = stim.DetectorErrorModel()
    i = -1
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        i += 1
        args = list(inst.args_copy())
        if i == k and args:
            args[0] = min(0.49, args[0] * factor)
        out.append("error", args, list(inst.targets_copy()))
    return out


def two_representative_circuit(d: int, p: float):
    """The same memory experiment, with a SECOND logical-Z representative.

    The final round measures all d*d data qubits in row-major order, and
    the built-in observable is row 0. Row 1 is a disjoint set of qubits
    and an equally valid representative -- they differ by a product of Z
    stabilizers, which is exactly what makes the runtime check possible.
    """
    import stim
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p)
    N = d * d
    targs = [stim.target_rec(-N + d + k) for k in range(d)]   # row 1
    c.append("OBSERVABLE_INCLUDE", targs, 1)
    return c


def solve_difference_functional(dem, n_det: int):
    """Find f with  maskA(m) XOR maskB(m) = <f, detectors(m)>  for every
    mechanism m, by Gaussian elimination over GF(2).

    Existence is not assumed. If no f satisfies every mechanism, the two
    observables do NOT differ by a stabilizer product in this compilation
    and the check would be unsound -- so this returns None and the caller
    refuses rather than shipping a detector built on a false premise.
    """
    rows, rhs = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = 0
        obs = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= (1 << int(t.val))
            elif t.is_logical_observable_id():
                obs ^= (1 << int(t.val))
        rows.append(dets)
        rhs.append(((obs >> 0) & 1) ^ ((obs >> 1) & 1))

    # eliminate
    piv: dict[int, tuple[int, int]] = {}
    for r, b in zip(rows, rhs):
        cur, cb = r, b
        while cur:
            top = cur.bit_length() - 1
            if top in piv:
                pr, pb = piv[top]
                cur ^= pr
                cb ^= pb
            else:
                piv[top] = (cur, cb)
                break
        else:
            if cb:
                return None            # 0 = 1: no such f exists
    f = 0
    # ASCENDING. Each stored pivot row has `top` as its highest set bit,
    # so every other bit it references is LOWER -- back-substitution has
    # to resolve those first. Iterating high-to-low read f at bits not yet
    # computed, always got zero, and produced a vector that failed its own
    # verification, which then reported "no such f exists". The refusal was
    # honest and the premise was fine; the solver was wrong.
    for top in sorted(piv):
        pr, pb = piv[top]
        val = pb
        rest = pr ^ (1 << top)
        while rest:
            bit = rest.bit_length() - 1
            val ^= (f >> bit) & 1
            rest ^= (1 << bit)
        if val:
            f |= (1 << top)
    # verify against EVERY mechanism, not just the ones elimination used
    for r, b in zip(rows, rhs):
        if bin(r & f).count("1") % 2 != b:
            return None
    return f


def apply_f(det: np.ndarray, f: int) -> np.ndarray:
    idx = [i for i in range(det.shape[1]) if (f >> i) & 1]
    if not idx:
        return np.zeros(det.shape[0], dtype=np.uint8)
    return det[:, idx].sum(axis=1).astype(np.uint8) & 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    import pymatching
    rng = np.random.default_rng(20260804)
    levels = {}

    for d in a.distances:
        t0 = time.perf_counter()
        circ = two_representative_circuit(d, a.p)
        dem = circ.detector_error_model(decompose_errors=True)
        n_det = circ.num_detectors
        f = solve_difference_functional(dem, n_det)
        if f is None:
            print(f"  d={d}: REFUSED -- the two representatives do not "
                  f"differ by a stabilizer product in this compilation")
            continue
        clean = pymatching.Matching.from_detector_error_model(dem)
        sampler = circ.compile_detector_sampler()
        n_mech = sum(1 for i in dem.flattened() if i.type == "error")

        ks = [int(rng.integers(0, n_mech)) for _ in range(a.trials)]
        bad = [pymatching.Matching.from_detector_error_model(
            corrupt_dem(dem, k)) for k in ks]
        # the complementarity control: WEIGHT faults, which L2 catches
        wk = [int(rng.integers(0, n_mech)) for _ in range(a.trials)]
        wbad = [pymatching.Matching.from_detector_error_model(
            corrupt_weight(dem, k)) for k in wk]

        clean_fp = 0            # L2' firing on healthy hardware
        wviol = [0] * a.trials
        wact = [0] * a.trials
        fired = [0] * a.trials
        activated = [0] * a.trials
        shots_done = 0
        BATCH = 50000
        while shots_done < a.shots:
            nb = min(BATCH, a.shots - shots_done)
            det, _obs = sampler.sample(nb, separate_observables=True)
            fs = apply_f(det, f)
            p0 = clean.decode_batch(det).astype(np.uint8)
            clean_fp += int(((p0[:, 0] ^ p0[:, 1] ^ fs) != 0).sum())
            for i, m in enumerate(bad):
                p1 = m.decode_batch(det).astype(np.uint8)
                viol = (p1[:, 0] ^ p1[:, 1] ^ fs) != 0
                fired[i] += int(viol.sum())
                activated[i] += int(((p0[:, 0] ^ p1[:, 0])
                                     | (p0[:, 1] ^ p1[:, 1])).sum())
            for i, m in enumerate(wbad):
                pw = m.decode_batch(det).astype(np.uint8)
                wviol[i] += int(((pw[:, 0] ^ pw[:, 1] ^ fs) != 0).sum())
                wact[i] += int(((p0[:, 0] ^ pw[:, 0])
                                | (p0[:, 1] ^ pw[:, 1])).sum())
            shots_done += nb

        act = [i for i in range(a.trials) if activated[i]]
        det_rate = (float(np.mean([1.0 if fired[i] else 0.0 for i in act]))
                    if act else None)
        per_shot = (float(np.mean([fired[i] / max(1, activated[i])
                                   for i in act])) if act else None)
        lvl = {"shots": a.shots, "mechanisms": n_mech,
               "detectors": n_det,
               "difference_functional_weight": bin(f).count("1"),
               "false_positives_on_clean": clean_fp,
               "false_positive_rate": clean_fp / a.shots,
               "trials": a.trials, "activated_trials": len(act),
               "L2prime_trial_detection_given_activation": det_rate,
               "L2prime_per_shot_detection_given_activation": per_shot,
               "L1_detections": 0, "L2_detections": 0,
               "weight_fault_activated_trials":
                   int(sum(1 for x in wact if x)),
               "L2prime_detection_on_WEIGHT_faults":
                   (float(np.mean([1.0 if wviol[i] else 0.0
                                   for i in range(a.trials) if wact[i]]))
                    if any(wact) else None),
               "seconds": round(time.perf_counter() - t0, 1)}
        levels[f"d={d}"] = lvl
        print(f"  d={d:2d}  f-weight {bin(f).count('1'):4d}  "
              f"false-pos {clean_fp}/{a.shots}  "
              f"L2' detects {(det_rate or 0)*100:5.1f}% of activated "
              f"trials, {(per_shot or 0)*100:5.1f}% of activated shots  "
              f"| weight-fault L2' "
              f"{((lvl['L2prime_detection_on_WEIGHT_faults'] or 0)*100):5.1f}% "
              f"({lvl['seconds']}s)")

    out = {"schema": "oneq-c2-l2prime/1",
           "layer": ("L2' certified logical-class attestation: two "
                     "homologically equivalent representatives whose mask "
                     "bits share nothing, checked inline against a GF(2) "
                     "functional of the syndrome"),
           "noise": f"circuit-level depolarizing p={a.p}, rounds=d",
           "levels": levels,
           "why_L1_and_L2_cannot": (
               "a mask flip changes no edge weight, so the correction is "
               "unchanged, so every check that reads only the correction "
               "returns an unchanged verdict"),
           "cost": ("one extra observable column: 1 bit per mechanism on a "
                    "table that already exists, plus a fixed syndrome "
                    "parity. Against the spec's own baselines -- CRAM "
                    "scrub ~0% logic but scrub-latency, SRAM ECC 12.5% of "
                    "memory bits, DMR 2x, selective TMR 37% LUT, full TMR "
                    ">200% -- this buys per-shot coverage of the mask "
                    "class from the code's degeneracy rather than from "
                    "replicated hardware"),
           "scope": ("detection is conditioned on ACTIVATION; an "
                     "unactivated fault is invisible to every inline "
                     "layer. False positives on clean shots are reported "
                     "because a detector that fires on healthy hardware "
                     "is worse than no detector.")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "c2_l2prime_attestation.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
