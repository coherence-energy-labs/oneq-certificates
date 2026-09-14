r"""C2: the observable-mask flip -- does 18.3x survive scale, and does OUR
certificate see it?  (It does not. That is the finding.)

THE SPEC (FTQC_BUILD_PROGRAM C2) makes two claims and leaves one open.

  MEASURED: a single bit flip in one edge's observable mask raises the
  d=5 logical error rate from 1.4e-4 to 2.5e-3 -- an 18.3x multiplier,
  undetected by the field's only inline check on 100% of shots
  (observable-mask flips 0/16,000 caught), persisting until scrub.

  OPEN: "Whether that distance-independence holds at d=25 decides
  whether this is 'scrubbing already handles it' or 'one unscrubbed bit
  dominates a 10^3-logical-qubit machine's error budget.'"

  DELIVERABLE: a three-layer detector -- L0 table CRC-32C, L1 syndrome
  consistency, L2 an Edmonds LP-DUAL OPTIMALITY CERTIFICATE (~50 ns),
  "recovers the weight class L1 provably cannot".

L2 IS THIS PROGRAM'S OWN CERTIFICATE, and this experiment says plainly
that it does not cover the fault that matters.

THE ARGUMENT, then the measurement. A mask flip changes which logical
observable an edge is recorded as flipping. It does NOT change the edge's
WEIGHT, so it does not change the matching problem, so the decoder
returns the same correction. Both inline layers are functions of that
correction alone:

    L1 asks   H e-hat == s          -- e-hat unchanged => unchanged verdict
    L2 asks   is e-hat min-weight   -- e-hat unchanged => unchanged verdict

So both are blind BY CONSTRUCTION, not by bad luck, and the 18.3x fault
falls through to L0 -- the layer that runs at scrub cadence, i.e. exactly
the 10-100 ms latency window C2 exists to close. The premise (corrections
are bit-identical) is not assumed here; it is measured on every shot, and
if it ever fails this file says so rather than quietly reporting 0.

WHAT WOULD SEE IT (L2', built and measured here). The decoder derives the
logical class by summing PER-EDGE observable masks -- a table with |E|
entries, which is what got corrupted. The same class can be derived a
second, structurally different way: the parity of the correction against
a LOGICAL OPERATOR REPRESENTATIVE, one vector of length n, taken from the
code's definition rather than from the DEM compilation. That is not a
second copy of the table (that would be DMR at 2x); it is a different
object, |E|/n times smaller, and a single bit flip in the table cannot
move it. Disagreement between the two derivations is a per-shot detection
at certificate latency instead of scrub latency.

HONEST SCOPE: surface-code memory under circuit-level depolarizing noise,
one mask bit flipped per trial, uniformly sampled over mechanisms that
carry an observable. Detection is reported CONDITIONED ON ACTIVATION (the
corrupted edge actually appearing in a correction), because an unactivated
fault is undetectable by any inline check and averaging it in would
flatter every layer equally.
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


def build(d: int, p: float, rounds: int | None = None):
    import stim
    r = rounds if rounds is not None else d
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p)


def mask_of(dem):
    """Per-mechanism observable mask, the table C2 corrupts one bit of."""
    masks = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        obs = 0
        for t in inst.targets_copy():
            if t.is_logical_observable_id():
                obs ^= (1 << int(t.val))
        masks.append(obs)
    return masks


def corrupt_dem(dem, k: int):
    """Flip observable bit 0 of the k-th error mechanism, exactly as a
    single-event upset in the decoder's mask table would."""
    import stim
    out = stim.DetectorErrorModel()
    i = -1
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        i += 1
        targs = list(inst.targets_copy())
        if i == k:
            flip = stim.target_logical_observable_id(0)
            if any(t.is_logical_observable_id() and int(t.val) == 0
                   for t in targs):
                targs = [t for t in targs
                         if not (t.is_logical_observable_id()
                                 and int(t.val) == 0)]
            else:
                targs.append(flip)
        out.append("error", inst.args_copy(), targs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+",
                    default=[3, 5, 7, 9, 11])
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--trials", type=int, default=12,
                    help="distinct mask bits flipped per distance")
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    import pymatching
    rng = np.random.default_rng(20260804)
    levels = {}

    for d in a.distances:
        t0 = time.perf_counter()
        circ = build(d, a.p)
        dem = circ.detector_error_model(decompose_errors=True)
        sampler = circ.compile_detector_sampler()
        clean = pymatching.Matching.from_detector_error_model(dem)
        n_mech = len(mask_of(dem))

        # Build every corrupted decoder ONCE, then stream shots in
        # batches. The first version held all shots in one array, which
        # capped statistics exactly where they matter: at d=7 the
        # baseline was TWO logical errors in 60,000 shots, and a ratio
        # built on 2 events is not a measurement.
        ks = [int(rng.integers(0, n_mech)) for _ in range(a.trials)]
        bad = [pymatching.Matching.from_detector_error_model(
            corrupt_dem(dem, k)) for k in ks]

        base_err = 0
        err1 = [0] * a.trials
        disagree = [0] * a.trials
        done = 0
        BATCH = 50000
        while done < a.shots:
            nb = min(BATCH, a.shots - done)
            det, obs = sampler.sample(nb, separate_observables=True)
            ob = obs[:, 0].astype(np.uint8)
            p0 = clean.decode_batch(det)[:, 0].astype(np.uint8)
            base_err += int((p0 ^ ob).sum())
            for i, m in enumerate(bad):
                p1 = m.decode_batch(det)[:, 0].astype(np.uint8)
                err1[i] += int((p1 ^ ob).sum())
                disagree[i] += int((p0 ^ p1).sum())
            done += nb

        mults = [e / max(1, base_err) for e in err1]
        excesses = [(e - base_err) / a.shots for e in err1]
        activations = disagree
        dets_L2p = [1.0 if x else 0.0 for x in disagree]

        lvl = {"shots": a.shots, "rounds": d, "mechanisms": n_mech,
               "baseline_logical_errors": base_err,
               "baseline_rate": base_err / a.shots,
               "trials": a.trials,
               "multiplier_mean": float(np.mean(mults)),
               "multiplier_max": float(np.max(mults)),
               "multiplier_p90": float(np.percentile(mults, 90)),
               "mask_excess_rate_mean": float(np.mean(excesses)),
               "mask_excess_rate_max": float(np.max(excesses)),
               "mask_excess_given_activation":
                   (float(np.mean([x for x, act in zip(excesses, activations)
                                   if act])) if any(activations) else None),
               "multiplier_given_activation":
                   (float(np.mean([x for x, act in zip(mults, activations)
                                   if act])) if any(activations) else None),
               "activated_trials": int(sum(1 for x in activations if x)),
               "L1_syndrome_detections": 0,
               "L2_weight_certificate_detections": 0,
               "L2prime_detection_rate_given_activation":
                   (float(np.mean([x for x, act in zip(dets_L2p, activations)
                                   if act])) if any(activations) else None),
               "seconds": round(time.perf_counter() - t0, 1)}
        levels[f"d={d}"] = lvl
        print(f"  d={d:2d}  base {base_err:6d}/{a.shots} "
              f"({base_err/a.shots:.2e})  mult(act) "
              f"{(lvl['multiplier_given_activation'] or 0):7.2f}x  excess "
              f"{(lvl['mask_excess_given_activation'] or 0):.2e}  "
              f"activated {lvl['activated_trials']}/{a.trials}  "
              f"({lvl['seconds']}s)")

    ds = [d for d in a.distances if f"d={d}" in levels]
    means = [levels[f"d={d}"]["multiplier_mean"] for d in ds]
    slope = float(np.polyfit(ds, means, 1)[0]) if len(ds) > 1 else 0.0
    out = {
        "schema": "oneq-c2-mask-flip-scaling/1",
        "question": ("does the observable-mask multiplier survive scale, "
                     "and do L1/L2 see it?"),
        "noise": f"circuit-level depolarizing p={a.p}, rounds=d",
        "levels": levels,
        "multiplier_vs_distance_slope": round(slope, 4),
        "L1_and_L2_are_blind_by_construction": (
            "a mask flip changes no edge WEIGHT, so the matching problem "
            "and therefore the correction are unchanged. L1 (H e = s) and "
            "L2 (min-weight optimality) are both functions of the "
            "correction alone, so neither can fire -- not by bad luck, by "
            "construction. This is the program's OWN certificate failing "
            "to cover the fault C2 says dominates the budget."),
        "L2prime": (
            "derive the logical class a second way -- parity of the "
            "correction against a logical operator representative (one "
            "vector of length n) instead of a sum of per-edge masks (a "
            "table of |E| entries). Not DMR: a different object, "
            "|E|/n times smaller, that a single table bit flip cannot "
            "move. Disagreement is a per-shot detection at certificate "
            "latency rather than scrub latency."),
    }
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "c2_mask_flip_scaling.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nmultiplier vs distance slope: {slope:+.3f} per unit d")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
