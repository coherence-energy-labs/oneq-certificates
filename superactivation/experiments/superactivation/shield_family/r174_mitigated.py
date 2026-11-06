"""R174: R146's and R147's F_e with the TERMINAL readout inverted -- a SECONDARY estimate, never certificate-grade
(item 23 of the owner's 25-item list).

The claim is on RAW counts (the certificate's own rule). Readout is the largest single loss at m = 5 (0.050 of 0.131).
Whether the final readout may be corrected in a claim is the owner's contract decision (docs/OPEN_WORK.md). This
gives the number that decision is about, computed the honest way:
- the per-qubit confusion is R152's OWN in-circuit readout family (the repaired snapshots m146 / m147);
- for each unit, the score is replaced by g = (x_q A_q^{-T}) s over the unit's scored bits, so E_obs[g] = E_true[s]
  when the readout errors are independent per qubit (A_q = [[1-p01, p10], [p01, 1-p10]], rows observed);
- the units are then aggregated EXACTLY as the measured estimate aggregates them (hw_superact_run.estimate for
  R146's p0; R147's control-variate estimator for its main units).
Mid-circuit reads are NOT corrected: their errors drive physical feed-forward, which no post-processing undoes.
Independence of readout errors across qubits is an assumption; R152 found no readout crosstalk at the bit level, and
R169 looks at the IQ level.

  python r174_mitigated.py REPO_ROOT -> evidence/superactivation/r174_mitigated/mitigated_r152.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))


class MitigationRefused(ValueError):
    """the correction cannot be applied faithfully"""


def confusion(p01: float, p10: float) -> np.ndarray:
    """rows: observed 0/1; columns: true 0/1"""
    A = np.array([[1 - p01, p10], [p01, 1 - p10]], dtype=float)
    if not (0 <= p01 < 0.5 and 0 <= p10 < 0.5):
        raise MitigationRefused(f"a confusion with p01 {p01}, p10 {p10} is not invertible as a readout")
    return A


def corrected_score(score, confusions: list) -> np.ndarray:
    """g over all outcomes, indexed g[b0, b1, ...] in clbit order: E_obs[g] = E_true[score] for independent readout"""
    m = len(confusions)
    s = np.empty((2,) * m)
    for bits in itertools.product((0, 1), repeat=m):
        s[bits] = float(score(bits))
    g = s
    for axis, A in enumerate(confusions):
        Ainv_T = np.linalg.inv(A).T
        g = np.moveaxis(np.tensordot(Ainv_T, np.moveaxis(g, axis, 0), axes=(1, 0)), 0, axis)
    return g


def mean_of(counts: dict, g: np.ndarray, n_bits: int, bits_of) -> tuple:
    """(mean, variance per shot) of g over a unit's counts"""
    n = s1 = s2 = 0.0
    for key, c in counts.items():
        b = bits_of(key, n_bits)
        v = float(g[tuple(b[:g.ndim])])
        n += c
        s1 += c * v
        s2 += c * v * v
    if n <= 0:
        raise MitigationRefused("a unit with no shots")
    mean = s1 / n
    return mean, max(s2 / n - mean * mean, 0.0)


def scored_qubits(qc, n_bits: int) -> list:
    """the physical qubit behind each of the unit's first n_bits clbits"""
    by_clbit = {}
    for inst in qc.data:
        if inst.operation.name == "measure":
            by_clbit[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
    missing = [c for c in range(n_bits) if c not in by_clbit]
    if missing:
        raise MitigationRefused(f"scored clbits {missing} are written by no measurement")
    return [by_clbit[c] for c in range(n_bits)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    import hw_dynamic_run as R147
    import hw_faithful as FA
    import hw_superact_run as HS
    import hw_two_use as H
    import r152_posthoc as P152
    import r155_heldout_r152 as HO

    from oneq.hw import freeze as F
    from oneq.hw import stats as T
    art = P152.load(root / HO.R152_DIR)
    d = art["dir"]
    P = P152.predictors(art, 1)
    out: dict = dict(schema="oneq.r174.mitigated/1",
                     label="SECONDARY -- terminal readout inverted; the claim is on raw counts", r152_job=art["job"])
    # R146 p0
    r146 = art["parts"]["r146"]
    c146 = F.load(r146, (d / "r146.qpy").read_bytes(), check_sources=False)
    m6 = P["m146"]
    means_raw, means_mit, var_mit, shots = {}, {}, {}, {}
    for u in r146["units"]:
        score = H.score(r146["m"], HS.unit_from(u["unit"]))
        cnt = art["counts"]["r146__" + u["name"]]
        qs = scored_qubits(c146[u["name"]], u["n_bits"])
        g = corrected_score(score, [confusion(m6.p01[q], m6.p10[q]) for q in qs])
        ident = corrected_score(score, [np.eye(2)] * len(qs))
        means_raw[u["name"]] = mean_of(cnt, ident, u["n_bits"], T.bits_of)[0]
        means_mit[u["name"]], var_mit[u["name"]] = mean_of(cnt, g, u["n_bits"], T.bits_of)
        shots[u["name"]] = sum(cnt.values())
    raw146 = HS.estimate(means_raw, r146["units"])["F_e"]
    mit146 = HS.estimate(means_mit, r146["units"], var_mit, shots)
    out["r146"] = dict(raw_reproduced=raw146, raw_recorded=P["meas146"], mitigated=mit146["F_e"],
                       mitigated_shot_sd=mit146.get("shot_sd_F_e"), gain=mit146["F_e"] - raw146)
    # R147
    r147 = art["parts"]["r147"]
    c147 = F.load(r147, (d / "r147.qpy").read_bytes(), check_sources=False)
    m7 = P["m147"]
    means_raw, means_mit, var_mit, shots = {}, {}, {}, {}
    for u in r147["units"]:
        uu = R147.unit_from(u["unit"])
        score = R147._score(u)
        cnt = art["counts"]["r147__" + u["name"]]
        qs = scored_qubits(c147[u["name"]], u["n_bits"])
        g = corrected_score(score, [confusion(m7.p01[q], m7.p10[q]) for q in qs])
        ident = corrected_score(score, [np.eye(2)] * len(qs))
        means_raw[u["name"]] = mean_of(cnt, ident, u["n_bits"], T.bits_of)[0]
        means_mit[u["name"]], var_mit[u["name"]] = mean_of(cnt, g, u["n_bits"], T.bits_of)
        shots[u["name"]] = sum(cnt.values())
    main_units = [u for u in r147["units"] if u["kind"] == "main"]
    raw147 = R147.control_variate(means_raw, main_units)["F_e"]
    mit147 = R147.control_variate(means_mit, main_units, var_mit, shots)
    out["r147"] = dict(raw_reproduced=raw147, raw_recorded=P["meas147"], mitigated=mit147["F_e"],
                       mitigated_shot_sd=mit147.get("sd_F_e"), gain=mit147["F_e"] - raw147)
    for k in ("r146", "r147"):
        v = out[k]
        print(f"{k}: raw {v['raw_reproduced']:.4f} (recorded {v['raw_recorded']:.4f})  mitigated {v['mitigated']:.4f}"
              f" +- {v['mitigated_shot_sd'] or float('nan'):.4f}  gain {v['gain']:+.4f}", flush=True)
    od = root / "evidence" / "superactivation" / "r174_mitigated"
    od.mkdir(parents=True, exist_ok=True)
    (od / "mitigated_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                            newline="\n")
    print("wrote mitigated_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
