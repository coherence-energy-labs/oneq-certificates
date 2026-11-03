r"""Why the UNSCALED min-sum BP-OSD of paper 2 depends on the declared error rate, and the scaled one does not.

    python experiments/m0_prior_art/qldpc_minsum_rate_invariance.py

WHY (paper 2, second review round, 2026-11-02). qldpc_bposd_scaling_sensitivity.py measured, on the same 200 strong-
baseline syndromes at p = 0.05, that configuration B with ldpc's default min-sum scaling factor 1 fails between 9 and
109 times depending on the error rate it is told, while with scaling 0.625 it fails 9 times at every declared rate.

THE ARGUMENT. With a uniform prior every input of min-sum BP is +-L, L = log((1-p)/p), and every update is homogeneous
of degree 1 (a sum, a minimum of magnitudes, a sign, a product by the scaling factor); the OSD stage orders bits by
their soft values. So in EXACT arithmetic the decoder's output does not depend on the declared rate at all. Any
dependence is floating-point. At scaling 1 every message is, in exact arithmetic, an integer multiple of L, so
cancellations to exactly zero are common, and in floating point those zeros come out as rounding residues whose sign
depends on the bits of L.

PREDICTIONS, fixed before this ran:
  (1) at declared rates whose L are exact powers of two (1, 2, 4, 8: multiplication by 2 is exact in IEEE doubles),
      every answer is identical, shot for shot, at scaling 1 AND at scaling 0.625;
  (2) at scaling 1, across the sensitivity scan's rates, many shots change answer; at 0.625, few or none;
  (3) at scaling 1 the final soft values hold many rounding residues (0 < |LLR| < 1e-9 L) and exact zeros; at 0.625,
      far fewer.
Prediction (1) failing would mean the decoder does something that does not scale (a clip, a constant, an exp), and
the rounding story is wrong.

WHAT (1)-(3) GAVE, and the prediction added AFTER them. (1) held, 200/200, at both scaling factors; (2) held (scaling 1:
85-88 of 200 shots keep one answer across the scan; 0.625: 199-200). (3) FAILED as worded: there are NO rounding
residues near zero at either factor; at scaling 1 there are instead ~6,300 EXACT zeros (about 31 bits per shot), at
0.625 none. So the rate enters through TIES, not residues: at scaling 1 the soft values are integer multiples of L, many
bits tie exactly, and the order the OSD stage gives tied bits can turn on the last bit of sums that are equal in exact
arithmetic. Added then, and so not a blind prediction:
  (4) a fixed relative jitter of 1e-6 in the prior (L_i = L (1 + 1e-6 u_i), u_i fixed; still homogeneous in L, so the
      exact-arithmetic invariance is kept) breaks the ties: at scaling 1 the answers become the same at every scanned
      rate on (almost) every shot, and exact ties in the final soft values (bits sharing a magnitude) vanish.

The strong-baseline v3 shots (evidence/strong_baseline_v3); configurations A and B of qldpc_strong_baseline_gate.
Post hoc; nothing here was preregistered.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "qldpc_minsum_rate_invariance.json"
V3 = ROOT / "evidence" / "strong_baseline_v3"
SENS = ROOT / "evidence" / "qldpc_bposd_scaling_sensitivity.json"
POW2 = [1.0, 2.0, 4.0, 8.0]
ALPHAS = [1.0, 0.625]
LEVEL = "0.05"


def rate_for_llr(L: float) -> float:
    """A double p with log((1-p)/p) == L exactly (the decoder's own formula, checked below)."""
    p = 1.0 / (1.0 + math.exp(L))
    for _ in range(64):
        got = math.log((1 - p) / p)
        if got == L:
            return p
        p = math.nextafter(p, 0.0 if got < L else 1.0)
    raise SystemExit(f"no double rate gives prior LLR exactly {L}")


def main() -> int:
    from ldpc import BpOsdDecoder
    import ldpc
    from scipy.sparse import csr_matrix
    from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
    from qldpc_strong_baseline_gate import CONFIGS

    HX, HZ = bb_gross_code()
    n = HZ.shape[1]
    xb, xp = gf2_row_reduce(HX % 2)
    shots = [json.loads(x) for x in (V3 / f"p{LEVEL}_shots.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    sens = json.loads(SENS.read_text(encoding="utf-8"))
    scan_rates = [float(r) for r in sens["declared_rate_scan"][f"p={LEVEL} strong_100_osdcs10"]]

    # the decoder's prior is log((1-p)/p), bit for bit: read it back through a two-check toy whose middle bit
    # receives exactly one cancelling message (+L - L + L)
    toy = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    for L in POW2:
        p = rate_for_llr(L)
        d = BpOsdDecoder(toy, error_rate=p, max_iter=5, bp_method="ms", ms_scaling_factor=1.0, osd_method="osd_cs",
                         osd_order=0)
        d.decode(np.array([1, 0], dtype=np.uint8))
        if float(d.log_prob_ratios[1]) != L:
            raise SystemExit(f"the decoder's prior at p={p!r} is {d.log_prob_ratios[1]!r}, not {L}")

    jitter = np.random.default_rng(20261102).random(n)

    def run(cfg: dict, alpha: float, rate: float, jittered: bool = False) -> dict:
        L = math.log((1 - rate) / rate)
        if jittered:
            Li = L * (1.0 + 1e-6 * jitter)
            dec = BpOsdDecoder(csr_matrix(HZ), error_channel=list(1.0 / (1.0 + np.exp(Li))), ms_scaling_factor=alpha,
                               **cfg)
        else:
            dec = BpOsdDecoder(csr_matrix(HZ), error_rate=rate, ms_scaling_factor=alpha, **cfg)
        answers, fails, residues, zeros, converged, tied = [], 0, 0, 0, 0, 0
        for s in shots:
            syn = np.zeros(HZ.shape[0], dtype=np.uint8)
            syn[s["syndrome_support"]] = 1
            e = np.asarray(dec.decode(syn)).astype(np.uint8)
            answers.append(tuple(int(i) for i in np.flatnonzero(e)))
            lpr = np.abs(np.asarray(dec.log_prob_ratios, dtype=float))
            zeros += int((lpr == 0).sum())
            residues += int(((lpr > 0) & (lpr < 1e-9 * L)).sum())
            _, counts = np.unique(lpr, return_counts=True)
            tied += int(counts[counts > 1].sum())          # bits whose magnitude another bit shares exactly
            converged += bool(dec.converge)
            t = np.zeros(n, dtype=np.uint8)
            t[s["true_error_support"]] = 1
            f = (e ^ t) % 2
            fails += bool(((HZ @ f) % 2).any()) or not gf2_in_rowspace(xb, xp, f, n)
        return {"answers": answers, "failures": fails, "lpr_exact_zeros": zeros, "lpr_rounding_residues": residues,
                "lpr_tied_bits": tied, "bp_converged": converged}

    out = {"schema": "oneq-qldpc-minsum-rate-invariance/1", "ldpc": ldpc.__version__, "post_hoc": "not preregistered",
           "level": LEVEL, "shots": len(shots), "pow2_llrs": POW2, "pow2_rates": [rate_for_llr(L) for L in POW2],
           "scan_rates": scan_rates, "configs": {}}
    for label in ("baseline_30_osdcs5", "strong_100_osdcs10"):
        for alpha in ALPHAS:
            pw = [run(CONFIGS[label], alpha, rate_for_llr(L)) for L in POW2]
            sc = {r: run(CONFIGS[label], alpha, r) for r in scan_rates}
            ref = sc[float(LEVEL)]["answers"]
            row = {
                "pow2_all_identical_shots": sum(len({pw[k]["answers"][i] for k in range(len(POW2))}) == 1
                                                for i in range(len(shots))),
                "pow2_failures": [x["failures"] for x in pw],
                "scan_shots_with_one_answer": sum(len({sc[r]["answers"][i] for r in scan_rates}) == 1
                                                  for i in range(len(shots))),
                "scan_shots_differing_from_true_rate": {str(r): sum(a != b for a, b in zip(sc[r]["answers"], ref))
                                                        for r in scan_rates},
                "scan_failures": {str(r): sc[r]["failures"] for r in scan_rates},
                "true_rate_lpr_exact_zeros": sc[float(LEVEL)]["lpr_exact_zeros"],
                "true_rate_lpr_rounding_residues": sc[float(LEVEL)]["lpr_rounding_residues"],
                "true_rate_lpr_tied_bits": sc[float(LEVEL)]["lpr_tied_bits"],
                "true_rate_bp_converged": sc[float(LEVEL)]["bp_converged"],
            }
            # (4): the same scan with the ties broken by a fixed relative jitter
            sj = {r: run(CONFIGS[label], alpha, r, jittered=True) for r in scan_rates}
            row["jittered"] = {
                "scan_shots_with_one_answer": sum(len({sj[r]["answers"][i] for r in scan_rates}) == 1
                                                  for i in range(len(shots))),
                "scan_failures": {str(r): sj[r]["failures"] for r in scan_rates},
                "true_rate_lpr_exact_zeros": sj[float(LEVEL)]["lpr_exact_zeros"],
                "true_rate_lpr_tied_bits": sj[float(LEVEL)]["lpr_tied_bits"],
                "true_rate_shots_differing_from_unjittered": sum(a != b for a, b in zip(sj[float(LEVEL)]["answers"],
                                                                                        ref)),
            }
            out["configs"][f"{label} alpha={alpha:g}"] = row
            print(f"{label} alpha={alpha:g}: pow2 identical {row['pow2_all_identical_shots']}/{len(shots)} "
                  f"(failures {row['pow2_failures']}); scan one-answer {row['scan_shots_with_one_answer']}/{len(shots)}; "
                  f"zeros {row['true_rate_lpr_exact_zeros']} residues {row['true_rate_lpr_rounding_residues']} "
                  f"tied {row['true_rate_lpr_tied_bits']}; jittered: one-answer "
                  f"{row['jittered']['scan_shots_with_one_answer']}/{len(shots)} tied "
                  f"{row['jittered']['true_rate_lpr_tied_bits']} failures "
                  f"{sorted(set(row['jittered']['scan_failures'].values()))}", flush=True)
    # the scan must reproduce the sensitivity artifact it explains
    for label in ("baseline_30_osdcs5", "strong_100_osdcs10"):
        mine = out["configs"][f"{label} alpha=1"]["scan_failures"]
        theirs = sens["declared_rate_scan"][f"p={LEVEL} {label}"]
        if {float(k): v for k, v in mine.items()} != {float(k): v for k, v in theirs.items()}:
            raise SystemExit(f"{label}: the scan does not reproduce qldpc_bposd_scaling_sensitivity.json")
        mine_s = out["configs"][f"{label} alpha=0.625"]["scan_failures"]
        theirs_s = sens["declared_rate_scan_scaled_0.625"][f"p={LEVEL} {label}"]
        if {float(k): v for k, v in mine_s.items()} != {float(k): v for k, v in theirs_s.items()}:
            raise SystemExit(f"{label}: the scaled scan does not reproduce qldpc_bposd_scaling_sensitivity.json")
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
